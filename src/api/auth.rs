use anyhow::Result;
use jsonwebtoken::{EncodingKey, Header};
use salvo::{
    http::cookie::time::{Duration, OffsetDateTime},
    oapi::extract::JsonBody,
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::{config::ListenerOption, JwtClaims};

#[derive(Serialize, Deserialize, Debug, ToParameters, ToSchema)]
#[salvo(extract(
    default_source(from = "query"),
    default_source(from = "param"),
    default_source(from = "body"),
))]
struct AppKeyRequest {
    pub app_id: String,
    pub app_secret: String,
}

#[derive(Serialize, Deserialize, Debug, ToSchema, ToResponse)]
struct AuthResponse {
    token: Option<String>,
    code: u16,
    message: Option<String>,
}

/// Exchange `appkey` and `app_secret` with the authorization token.
#[endpoint(
    responses(
        (status_code = 200, description = "Exchange the token successfully.", body = AuthResponse),
        (status_code = 400, description = "Bad request to call this method.", body = AuthResponse),
        (status_code = 403, description = "Forbidden access by provided appId with secretKey.", body = AuthResponse),
    )
)]
pub fn exchange(depot: &mut Depot, req: JsonBody<AppKeyRequest>, res: &mut Response) {
    let listen_option = depot.get::<ListenerOption>("listen").unwrap();
    let auth = req.0;
    if listen_option
        .app_keys
        .clone()
        .into_iter()
        .any(|p| p.app_id == auth.app_id.clone() && p.secret_key == auth.app_secret.clone())
    {
        let exp = OffsetDateTime::now_utc()
            + Duration::seconds(listen_option.expire_sec.unwrap_or(86400u32) as i64);
        let claim = JwtClaims {
            sid: auth.app_id,
            exp: exp.unix_timestamp(),
        };
        match jsonwebtoken::encode(
            &Header::default(),
            &claim,
            &EncodingKey::from_secret(listen_option.slot.clone().as_bytes()),
        ) {
            Ok(token) => {
                res.render(Json(AuthResponse {
                    token: Some(token),
                    code: StatusCode::OK.as_u16(),
                    message: Some("SUCCESS".to_string()),
                }));
            }
            Err(err) => {
                log::info!("Unable to encoding jwt_token: {}", err);
                res.status_code(StatusCode::BAD_REQUEST)
                    .render(Json(AuthResponse {
                        token: None,
                        code: StatusCode::BAD_REQUEST.as_u16(),
                        message: Some(err.to_string()),
                    }));
            }
        }
    } else {
        res.status_code(StatusCode::FORBIDDEN)
            .render(Json(AuthResponse {
                token: None,
                code: StatusCode::FORBIDDEN.as_u16(),
                message: Some("NO-Match AppId and SecretKey".to_string()),
            }));
    }
}
